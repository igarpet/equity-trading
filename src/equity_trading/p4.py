"""P4: Da, Liu & Schaumburg (2014) - is reversal stronger once industry moves and news are removed?

Protocol: discussions/News_Filtered_Reversal/001_claude_p4_news_filtered_reversal_preregistration.md

Primary news: SEC EDGAR 8-K filings with item 2.02 (earnings); window {f-1, f, f+1} sessions around the filing.
Secondary news: Alpaca / Benzinga articles tagging <= 3 symbols; a session is a news shock when its count is
>= 3 and >= 3 x the mean of the previous 60 sessions (>= 20 observed).
Daily arms R (raw), I (industry-adjusted: minus sector ETF), IN (I without earnings-window stocks) use the
P1 quintile machinery; the 15-minute arm masks S05 entries inside earnings windows.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

P4_VERSION = "0.1.0"
NY = "America/New_York"


@dataclass(frozen=True)
class P4Config:
    horizons: tuple = (21, 5)
    alpha: float = 0.025
    n_quantiles: int = 5
    skip: int = 1
    min_price: float = 5.0
    win_before: int = 1
    win_after: int = 1
    max_tags: int = 3
    shock_min: int = 3
    shock_mult: float = 3.0
    shock_window: int = 60
    shock_min_obs: int = 20
    blocks: tuple = (10, 20, 40)
    n_boot: int = 10_000
    n_placebo_daily: int = 200
    n_placebo_15m: int = 100
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


# =================================================================== SEC EDGAR
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/"


def _sec_headers() -> dict:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError('Set SEC_USER_AGENT in .env, e.g. SEC_USER_AGENT="Your Name you@example.com" (SEC fair-access rule).')
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def _sec_get(url: str, session, retries: int = 5):
    for k in range(retries):
        time.sleep(0.13)                                      # <= ~8 requests per second
        r = session.get(url, headers=_sec_headers(), timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        time.sleep(min(30, 2 ** k))
    raise RuntimeError(f"SEC {r.status_code} for {url}")


def sec_symbol(alpaca_symbol: str) -> str:
    return alpaca_symbol.replace(".", "-").upper()


def parse_8k_202(block: dict) -> pd.DataFrame:
    """Columnar EDGAR filings block -> rows of 8-K (and 8-K/A) filings whose items include 2.02."""
    n = len(block.get("form", []))
    if not n:
        return pd.DataFrame(columns=["filing_date", "acceptance", "form", "items"])
    df = pd.DataFrame({"form": block["form"], "filing_date": block["filingDate"],
                       "acceptance": block.get("acceptanceDateTime", [""] * n), "items": block.get("items", [""] * n)})
    m = df["form"].isin(["8-K", "8-K/A"]) & df["items"].fillna("").str.split(",").apply(lambda xs: "2.02" in [x.strip() for x in xs])
    return df[m].reset_index(drop=True)


def edgar_earnings(symbols: list[str], start: str = "2016-01-01", end: str = "2025-12-31", refresh: bool = False,
                   verbose: bool = True) -> pd.DataFrame:
    """[symbol, filing_date, acceptance, form] for 8-K item 2.02 filings. Cached per symbol under <data>/SEC/8k202/."""
    import requests
    from .data import vendor_dir
    d = vendor_dir("SEC") / "8k202"
    d.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    tmap_path = vendor_dir("SEC") / "company_tickers.json"
    if refresh or not tmap_path.exists():
        tmap_path.write_text(json.dumps(_sec_get(SEC_TICKERS, sess)))
    tmap = {v["ticker"].upper(): int(v["cik_str"]) for v in json.loads(tmap_path.read_text()).values()}
    out, missing, t0 = [], [], time.time()
    for k, s in enumerate(symbols, 1):
        p = d / f"{sec_symbol(s)}.parquet"
        if p.exists() and not refresh:
            df = pd.read_parquet(p)
        else:
            cik = tmap.get(sec_symbol(s))
            if cik is None:
                missing.append(s)
                continue
            js = _sec_get(f"{SEC_SUBMISSIONS}CIK{cik:010d}.json", sess)
            parts = [parse_8k_202(js["filings"]["recent"])] if js else []
            for f in (js or {}).get("filings", {}).get("files", []):
                if f.get("filingTo", "9999") >= start:
                    more = _sec_get(SEC_SUBMISSIONS + f["name"], sess)
                    if more:
                        parts.append(parse_8k_202(more))
            df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["filing_date", "acceptance", "form", "items"])
            df["cik"] = cik
            tmp = p.with_suffix(".parquet.tmp"); df.to_parquet(tmp, index=False); os.replace(tmp, p)
        df = df.assign(symbol=s)
        out.append(df[(df["filing_date"] >= start) & (df["filing_date"] <= end)])
        if verbose and (k % 50 == 0 or k == len(symbols)):
            print(f"  EDGAR {k}/{len(symbols)} symbols, {time.time() - t0:.0f}s", flush=True)
    ev = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "filing_date"])
    ev = ev.drop_duplicates(["symbol", "filing_date"])
    if verbose and missing:
        print(f"  no CIK for {len(missing)} symbols: {missing[:20]}")
    return ev


# =================================================================== Alpaca news
NEWS_URL = "https://data.alpaca.markets/v1beta1/news"


def fetch_news(symbols: list[str], start: str, end: str, session=None, max_retries: int = 6) -> pd.DataFrame:
    """[id, created_at (UTC), symbols ('|'-joined)] for all articles tagging any of `symbols` in [start, end]."""
    import requests
    from .alpaca import _credentials, _limiter
    session = session or requests.Session()
    params = {"symbols": ",".join(symbols), "start": start, "end": end, "limit": 50, "sort": "asc",
              "include_content": "false"}
    rows, token = [], None
    while True:
        if token:
            params["page_token"] = token
        for attempt in range(max_retries):
            _limiter.wait()
            r = session.get(NEWS_URL, headers=_credentials(), params=params, timeout=60)
            if r.status_code == 200:
                break
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt))
                continue
            raise RuntimeError(f"Alpaca news {r.status_code}: {r.text[:300]}")
        else:
            raise RuntimeError(f"Alpaca news: gave up after {max_retries} retries.")
        js = r.json()
        for a in js.get("news") or []:
            rows.append((a.get("id"), a.get("created_at"), "|".join(a.get("symbols") or [])))
        token = js.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(rows, columns=["id", "created_at", "symbols"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
    return df


def download_news(symbols: list[str], years: list[int], batch: int = 10, verbose: bool = True) -> pd.DataFrame:
    """Resumable: one parquet per (batch of symbols, year) under <data>/Alpaca/news/. Returns all articles (dedup by id)."""
    from .data import vendor_dir
    d = vendor_dir("Alpaca") / "news"
    d.mkdir(parents=True, exist_ok=True)
    syms = sorted(symbols)
    batches = [syms[i:i + batch] for i in range(0, len(syms), batch)]
    tasks = [(b, y) for y in years for b in batches]
    paths = [d / f"{y}_{b[0]}_{b[-1]}_{len(b)}.parquet" for b, y in tasks]
    todo = [i for i, p in enumerate(paths) if not p.exists()]
    if verbose:
        print(f"news: {len(tasks) - len(todo)}/{len(tasks)} batch-years cached; downloading {len(todo)}", flush=True)
    t0 = time.time()
    for k, i in enumerate(todo, 1):
        b, y = tasks[i]
        df = fetch_news(b, f"{y}-01-01T00:00:00Z", f"{y}-12-31T23:59:59Z")
        tmp = paths[i].with_suffix(".parquet.tmp"); df.to_parquet(tmp, index=False); os.replace(tmp, paths[i])
        if verbose and (k % 10 == 0 or k == len(todo)):
            el = time.time() - t0
            print(f"  {k}/{len(todo)} batch-years, {el / 60:.1f} min, ~{el / k * (len(todo) - k) / 60:.0f} min left", flush=True)
    allp = [pd.read_parquet(p) for p in paths if p.exists()]
    return pd.concat(allp, ignore_index=True).drop_duplicates("id") if allp else pd.DataFrame(columns=["id", "created_at", "symbols"])


def news_session(created_at: pd.Series, sessions: pd.DatetimeIndex) -> np.ndarray:
    """Index of the session each article belongs to: the first session whose 16:00 ET close is at or after it."""
    t = pd.DatetimeIndex(created_at).tz_convert(NY)
    eff = t.normalize().tz_localize(None) + pd.to_timedelta((t.hour * 60 + t.minute >= 16 * 60).astype(int), unit="D")
    return np.searchsorted(sessions.values, eff.values, side="left")


def news_counts(articles: pd.DataFrame, sessions: pd.DatetimeIndex, symbols: list[str], max_tags: int = 3) -> pd.DataFrame:
    """(sessions x symbols) count of firm-specific articles (<= max_tags tagged symbols)."""
    a = articles.copy()
    a["tags"] = a["symbols"].fillna("").str.split("|")
    a = a[a["tags"].str.len() <= max_tags]
    a["sess"] = news_session(a["created_at"], sessions)
    a = a[a["sess"] < len(sessions)].explode("tags")
    a = a[a["tags"].isin(set(symbols))]
    c = a.groupby(["sess", "tags"]).size().unstack(fill_value=0)
    out = c.reindex(index=range(len(sessions)), columns=symbols, fill_value=0).astype(np.int32)
    out.index = sessions
    return out


def news_shocks(counts: pd.DataFrame, cfg: P4Config) -> pd.DataFrame:
    base = counts.shift(1).rolling(cfg.shock_window, min_periods=cfg.shock_min_obs).mean()
    return (counts >= cfg.shock_min) & (counts >= cfg.shock_mult * base) & base.notna()


# =================================================================== event windows
def event_flags(events: pd.DataFrame, sessions: pd.DatetimeIndex, symbols: list[str], before: int, after: int) -> pd.DataFrame:
    """(sessions x symbols) bool: True on sessions f-before .. f+after around each event (f = session of the
    filing date, or the next session if that date is not a trading day)."""
    flags = np.zeros((len(sessions), len(symbols)), bool)
    col = {s: j for j, s in enumerate(symbols)}
    if len(events):
        ev = events[events["symbol"].isin(col)]
        f = np.searchsorted(sessions.values, pd.to_datetime(ev["filing_date"]).values, side="left")
        j = ev["symbol"].map(col).to_numpy()
        for off in range(-before, after + 1):
            i = f + off
            ok = (i >= 0) & (i < len(sessions))
            flags[i[ok], j[ok]] = True
    return pd.DataFrame(flags, index=sessions, columns=symbols)


def any_in_window(flags: pd.DataFrame, k: int) -> pd.DataFrame:
    """True at d if any session d-k+1 .. d is flagged."""
    return flags.astype(np.int8).rolling(k, min_periods=1).max().astype(bool)


def random_exclusion(excl: np.ndarray, pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per row, a random set of `pool` entries of the same size as the row's excluded set."""
    out = np.zeros(excl.shape, bool)
    for t in range(len(excl)):
        k = int((excl[t] & pool[t]).sum())
        if k:
            cand = np.flatnonzero(pool[t])
            out[t, rng.choice(cand, size=k, replace=False)] = True
    return out


# =================================================================== daily arms
def sort_returns(signal: pd.DataFrame, eligible: pd.DataFrame, ret: pd.DataFrame, cost: pd.DataFrame | None, h: int,
                 nq: int, skip: int, window) -> pd.DataFrame:
    """Quintile reversal (P1 machinery) of `signal` among `eligible` names; daily gross/cost/net over `window`."""
    from .p1 import ls_returns, overlap, reversal_weights
    lab = pd.DataFrame(np.where(eligible.to_numpy() & signal.notna().to_numpy(), 0, -1).astype(np.int8),
                       index=signal.index, columns=signal.columns)
    return ls_returns(overlap(reversal_weights(signal, lab, 0, nq), h), ret, cost, skip).loc[window]


def placebo_filter(signal, base_elig, excl, ret, h, cfg: P4Config, window, n: int, seed: int) -> np.ndarray:
    """Mean daily gross of (random-exclusion arm - unfiltered arm) under `n` random exclusions of the same size."""
    rng = np.random.default_rng(seed)
    ref = sort_returns(signal, base_elig, ret, None, h, cfg.n_quantiles, cfg.skip, window)["gross"]
    pool = (base_elig & signal.notna()).to_numpy()
    ex = excl.to_numpy()
    out = np.empty(n)
    for k in range(n):
        rnd = random_exclusion(ex, pool, rng)
        el = pd.DataFrame(pool & ~rnd, index=signal.index, columns=signal.columns)
        out[k] = float((sort_returns(signal, el, ret, None, h, cfg.n_quantiles, cfg.skip, window)["gross"] - ref).mean())
    return out


def gates_daily(f1_boot: dict, f1_p: float, f2_boot: dict, f3_boot: dict, net2x_mean: float, alpha: float) -> dict:
    f1 = all(v[1] > 0 for v in f1_boot.values()) and f1_p <= alpha
    f2 = all(v[1] > 0 for v in f2_boot.values())
    f3 = all(v[1] > 0 for v in f3_boot.values()) and net2x_mean > 0
    return {"F1_news_filter": f1, "F2_industry_adjustment": f2, "F3_economic": f3}


def verdict_daily(g: dict) -> dict:
    """g: {window: gates_daily(...)} for one horizon."""
    f1 = all(x["F1_news_filter"] for x in g.values())
    f2 = all(x["F2_industry_adjustment"] for x in g.values())
    f3 = all(x["F3_economic"] for x in g.values())
    rep = "REPLICATES" if (f1 and f2) else ("PARTIAL" if (f1 or f2) else "NO_REPLICATION")
    return {"replication_verdict": rep, "economic_verdict": "PROFITABLE_AFTER_COSTS" if f3 else "NOT_PROFITABLE_AFTER_COSTS"}


# =================================================================== 15-minute arm
def gates_15m(diff_ci: tuple, ranks: dict, net1: dict, net2: dict, boot_net: dict, alpha: float = 0.05) -> dict:
    q1 = diff_ci[1] > 0
    q2 = ranks["total_return"]["p_upper"] <= alpha and ranks["sharpe"]["p_upper"] <= alpha
    q3 = net1["total_return"] > 0 and net2["total_return"] > 0 and all(v[1] > 0 for v in boot_net.values())
    return {"Q1_outside_minus_inside": q1, "Q2_vs_random_exclusion": q2, "Q3_economic": q3, "pass": q1 and q2 and q3}


def synthetic_events(sessions: pd.DatetimeIndex, symbols: list[str], per_year: int = 4, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in symbols:
        n = int(per_year * len(sessions) / 252)
        for i in rng.choice(len(sessions), size=n, replace=False):
            rows.append((s, sessions[i].strftime("%Y-%m-%d")))
    return pd.DataFrame(rows, columns=["symbol", "filing_date"])


def synthetic_daily_news(n_per_sector: int = 40, years: list[int] | None = None, seed: int = 4,
                         rev: float = 0.15, jump: float = 0.04) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Offline daily panel: 4 sector ETFs (random-walk factors); stocks = factor + idiosyncratic part whose
    ordinary moves partly reverse 2 sessions later, plus permanent jumps on earnings dates (4 per year).
    Returns (close incl. ETFs, volume, ETF_OF, events). Industry adjustment and news filtering should both help."""
    years = years or [2016, 2017, 2018]
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    D = len(days)
    etfs = ["XLK", "XLF", "XLE", "XLU"]
    cols, etf_of, data = [], {}, {}
    for e in etfs:
        f = rng.standard_normal(D) * 0.012
        data[e] = f
        for j in range(n_per_sector):
            s = f"{e[-1]}{j:02d}"
            etf_of[s] = e
            cols.append(s)
            eps = rng.standard_normal(D) * 0.015
            r = np.zeros(D)
            for t in range(D):
                r[t] = eps[t] - (rev * eps[t - 2] if t >= 2 else 0.0)
            data[s] = f + r
    ev = synthetic_events(days, cols, per_year=4, seed=seed + 1)
    pos = {s: i for i, s in enumerate(days)}
    for s, d in zip(ev["symbol"], ev["filing_date"]):
        data[s][pos[pd.Timestamp(d)]] += jump * rng.choice([-1, 1])
    C = pd.DataFrame({k: 50 * np.exp(np.cumsum(v)) for k, v in data.items()}, index=days)
    V = pd.DataFrame(1e6, index=days, columns=C.columns)
    return C, V, etf_of, ev
